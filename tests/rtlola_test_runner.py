import json
from pathlib import Path
import subprocess
import platform
import sys

BUILD_VERSIONS = ["release", "debug"]
EXIT_FAILURE = 1


def build_path(base_dir, parts):
    path = base_dir
    for part in parts:
        path = path.joinpath(part)
    return path


def print_fail(message, end='\n'):
    sys.stdout.write('\x1b[1;31m' + message.strip() + '\x1b[0m' + end)


def print_pass(message, end='\n'):
    sys.stdout.write('\x1b[1;32m' + message.strip() + '\x1b[0m' + end)


def print_warn(message, end='\n'):
    sys.stdout.write('\x1b[1;33m' + message.strip() + '\x1b[0m' + end)


def print_info(message, end='\n'):
    sys.stdout.write('\x1b[1;34m' + message.strip() + '\x1b[0m' + end)


def print_bold(message, end='\n'):
    sys.stdout.write('\x1b[1;37m' + message.strip() + '\x1b[0m' + end)


def print_additional_trigger(message, count):
    sys.stdout.write('\x1b[1;31m"' + message.strip() + "\" : " + str(count) + ' (0 expected) \x1b[0m\n')


def print_trigger(message, expected, actual):
    if actual < expected:
        print_trigger_not_enough(trigger, expected, actual)
    else:
        print_trigger_too_many(trigger, expected, actual)


def print_trigger_not_enough(message, expected, actual):
    sys.stdout.write('"' + message.strip() + '\"\x1b[1;34m' + " : {} ({} expected)".format(actual, expected) + '\x1b[0m\n')


def print_trigger_too_many(message, expected, actual):
    sys.stdout.write('"' + message.strip() + '\"\x1b[1;31m' + " : {} ({} expected)".format(actual, expected) + '\x1b[0m\n')


running_on_windows = platform.system() == "Windows"
executable_name = "stream_lab.exe" if running_on_windows else "stream_lab"

# TODO get build_version from command line an check against BUILD_VERSIONS
build_version = "debug" # debug build is used during development so probably already build

repo_base_dir = Path(".").resolve().parent
rtlola_executable_path = repo_base_dir / "target" / build_version / executable_name
rtlola_executable_path_string = str(rtlola_executable_path)

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

test_dir = Path('.')
tests = [test_file for test_file in test_dir.iterdir() if test_file.is_file() and test_file.suffix == ".rtlola_test"]
return_code = 0
for test_file in tests:
    total_number_of_tests += 1
    print("========================================================================")
    print_bold(str(test_file))
    with test_file.open() as fd:
        test_json = json.load(fd)
        spec_file = build_path(repo_base_dir, test_json["spec_file"].split('/')[1:])
        input_file = build_path(repo_base_dir, test_json["input_file"].split('/')[1:])
        run_result = subprocess.run([rtlola_executable_path_string, "monitor", "--offline", "--stdout", "--verbosity", "outputs", str(spec_file), "--csv-in", str(input_file)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(repo_base_dir), universal_newlines=True)
        something_wrong = False
        if run_result.returncode == 0:
            lines = iter(run_result.stdout.split("\n"))
            # skip the initial output about needed memory etc.
            first_line = next(lines)
            triggers_in_output = dict()
            if not first_line.startswith("The specification uses at most"):
                print("Header changed")  # TODO

            # count triggers
            for line in lines:
                if line == "":
                    continue
                if line.startswith("Trigger: "):
                    trigger_warning = line[len("Trigger: "):]
                    triggers_in_output.setdefault(trigger_warning, 0)
                    triggers_in_output[trigger_warning] += 1
                # else:
                #     print("Unexpected line: {}".format(line))

            # print diff in triggers
            # TODO allow for specifying a tolerance in the JSON
            expected_triggers = list(test_json["triggers"].keys())
            trigger_names = list(set(list(triggers_in_output.keys()) + expected_triggers))
            trigger_names.sort()
            for trigger in trigger_names:
                if trigger in expected_triggers:
                    actual = triggers_in_output[trigger] if trigger in triggers_in_output else 0
                    expected = 0
                    if trigger in expected_triggers:
                        expected = test_json["triggers"][trigger]["expected_count"]
                    if actual != expected:
                        print_trigger(trigger, expected, actual)
                        something_wrong = True
                else:
                    print_additional_trigger(trigger, triggers_in_output[trigger])
                    something_wrong = True
            if something_wrong:
                wrong_tests += 1
        else:
            crashed_tests += 1
            print_fail("Returned with error code")
            something_wrong = True

        if something_wrong:
            if False:
                print("STDOUT")
                print(run_result.stdout)
            print("STDERR")
            print(run_result.stderr)
            print_fail("FAIL: " + str(test_file) + "\n" + test_json["rationale"])
            return_code = 1
        else:
            tests_passed +=1
            print_pass("PASS:" + str(test_file))

        print("")
print("========================================================================")
print("Total tests: {}".format(total_number_of_tests))
print_pass("Tests passed: {}".format(tests_passed))
print_fail("Tests crashed: {}".format(crashed_tests))
print_fail("Tests with wrong output: {}".format(wrong_tests))
print("")
print_bold("Passing rate: {:05.2f}%".format((100.0*tests_passed/total_number_of_tests)))
print("========================================================================")
sys.exit(return_code)