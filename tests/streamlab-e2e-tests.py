import json
from pathlib import Path
import subprocess
import platform
import sys
import argparse
import re


BUILD_VERSIONS = ["release", "debug"]
EXIT_FAILURE = 1



def build_path(base_dir, parts):
    path = base_dir
    for part in parts:
        path = path.joinpath(part)
    return path


def print_fail(message, end='\n'):
    sys.stdout.write('\x1b[1;31m' + message.rstrip() + '\x1b[0m' + end)


def print_pass(message, end='\n'):
    sys.stdout.write('\x1b[1;32m' + message.rstrip() + '\x1b[0m' + end)


def print_warn(message, end='\n'):
    sys.stdout.write('\x1b[1;33m' + message.rstrip() + '\x1b[0m' + end)


def print_info(message, end='\n'):
    sys.stdout.write('\x1b[1;34m' + message.rstrip() + '\x1b[0m' + end)


def print_bold(message, end='\n'):
    sys.stdout.write('\x1b[1;37m' + message.rstrip() + '\x1b[0m' + end)


def print_additional_trigger(message, count):
    sys.stdout.write('\x1b[1;31m"' + message.rstrip() + "\" : " + str(count) + ' (0 expected) \x1b[0m\n')


def print_trigger(message, expected, actual):
    if actual < expected:
        print_trigger_not_enough(trigger, expected, actual)
    else:
        print_trigger_too_many(trigger, expected, actual)


def print_trigger_not_enough(message, expected, actual):
    sys.stdout.write('"' + message.strip() + '\"\x1b[1;34m' + " : {} ({} expected)".format(actual, expected) + '\x1b[0m\n')


def print_trigger_too_many(message, expected, actual):
    sys.stdout.write('"' + message.strip() + '\"\x1b[1;31m' + " : {} ({} expected)".format(actual, expected) + '\x1b[0m\n')


parser = argparse.ArgumentParser(description='Run end-to-end tests for StreamLab')


running_on_windows = platform.system() == "Windows"
executable_name = "streamlab.exe" if running_on_windows else "streamlab"

# TODO get build_version from command line an check against BUILD_VERSIONS
# debug build is used during development so probably already build
build_version = "debug"

repo_base_dir = Path(".").resolve()
if not Path(".gitlab-ci.yml").exists():
    if (repo_base_dir.parent/".gitlab-ci.yml").exists():
        repo_base_dir = repo_base_dir.parent
    else:
        print_fail("Run this script from the repo base or from te tests directory!")
        sys.exit(EXIT_FAILURE)
streamlab_executable_path = repo_base_dir / "target" / build_version / executable_name
streamlab_executable_path_string = str(streamlab_executable_path)

if build_version == "debug":
    cargo_build = subprocess.run(["cargo", "build"], cwd=str(repo_base_dir))
else:
    cargo_build = subprocess.run(["cargo", "build", "--release"], cwd=str(repo_base_dir))
if cargo_build.returncode != 0:
    sys.exit(EXIT_FAILURE)

total_number_of_tests = 0
crashed_tests = 0
wrong_tests = 0
tests_passed = 0

test_dir = repo_base_dir/"tests"
tests = [test_file for test_file in test_dir.iterdir() if test_file.is_file() and test_file.suffix == ".streamlab_test"]
if len(sys.argv) == 2:
    tests = [test_file for test_file in tests if sys.argv[1] in test_file.name]

tests_passed = []
tests_crashed = []
tests_wrong_out = []
return_code = 0
for (mode, config) in [('interpreted', ["--interpreted"]), ('closure', []), ('time-info', ["--time-info-rep", "absolute"])]:
    check_time_info = "--time-info-rep" in config
    for test_file in tests:
        total_number_of_tests += 1
        print("========================================================================")
        test_name = "{} @ {}".format(mode, test_file.name.split('.')[0])
        print_bold("{}:".format(test_name))
        with test_file.open() as fd:
            test_json = json.load(fd)
            spec_file = build_path(repo_base_dir, test_json["spec_file"].split('/')[1:])
            input_file = build_path(repo_base_dir, test_json["input_file"].split('/')[1:])
            something_wrong = False
            run_result = None
            try:
                run_result = subprocess.run([streamlab_executable_path_string, "--offline", "--stdout", "--verbosity", "outputs", str(spec_file), "--csv-in", str(input_file)] + config, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(repo_base_dir), universal_newlines=True, timeout=10)
            except subprocess.TimeoutExpired:
                tests_crashed.append(test_name)
                print_fail("Test timed out")
                something_wrong = True
            if run_result is not None:
                if run_result.returncode == 0:
                    lines = iter(run_result.stdout.split("\n"))
                    triggers_in_output = dict()

                    # count triggers
                    for line in lines:
                        if line == "":
                            continue
                        m = re.match(r'((?P<timeinfo>.*): )?Trigger: (?P<trig_msg>.*)', line)
                        if m:
                            timeinfo = m.group('timeinfo')
                            trig_msg = m.group('trig_msg')
                            triggers_in_output.setdefault(trig_msg, [])
                            triggers_in_output[trig_msg].append(timeinfo)
                            continue
                        #print("Unexpected line: {}".format(line))

                    # print diff in triggers
                    # TODO allow for specifying a tolerance in the JSON
                    expected_triggers = list(test_json["triggers"].keys())
                    trigger_names = list(set(list(triggers_in_output.keys()) + expected_triggers))
                    trigger_names.sort()
                    for trigger in trigger_names:
                        if trigger in expected_triggers:
                            actual_time_info = triggers_in_output[trigger] if trigger in triggers_in_output else []
                            actual_count = len(actual_time_info)
                            expected_count = 0
                            expected_time_info = []
                            if trigger in expected_triggers:
                                expected_count = test_json["triggers"][trigger]["expected_count"]
                                expected_time_info = test_json["triggers"][trigger]["time_info"]
                                assert expected_count == len(expected_time_info)
                            if actual_count != expected_count:
                                print_trigger(trigger, expected_count, actual_count)
                                something_wrong = True
                            if check_time_info and actual_time_info != expected_time_info:
                                print_fail("time info for trigger \"{}\" incorrect".format(trigger))
                                #TODO give better info where the error is located
                                something_wrong = True
                        else:
                            print_additional_trigger(trigger, triggers_in_output[trigger])
                            something_wrong = True
                    if something_wrong:
                        tests_wrong_out.append(test_name)
                else:
                    tests_crashed.append(test_name)
                    print_fail("Returned with error code")
                    something_wrong = True

            if something_wrong:
                if False:
                    print("STDOUT")
                    print(run_result.stdout)
                    print("STDERR")
                    print(run_result.stderr)
                print_fail("FAIL")
                print_fail(test_json["rationale"])
                return_code = 1
            else:
                tests_passed.append(test_name)
                print_pass("PASS")

            print("")
print("========================================================================")
print("Total tests: {}".format(total_number_of_tests))
print_pass("Tests passed: {}".format(len(tests_passed)))
if len(tests_crashed) > 0:
    print_fail("Tests crashed: {}".format(len(tests_crashed)))
    for test in tests_crashed:
        print_fail("\t{}".format(test))
if len(tests_wrong_out) > 0:
    print_fail("Tests with wrong output: {}".format(len(tests_wrong_out)))
    for test in tests_wrong_out:
        print_fail("\t{}".format(test))
if total_number_of_tests > 0:
    print("")
    print_bold("Passing rate: {:.2f}%".format((100.0*len(tests_passed)/total_number_of_tests)))
print("========================================================================")
sys.exit(return_code)
