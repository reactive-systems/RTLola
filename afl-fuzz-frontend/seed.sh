#!/usr/bin/env bash

rm -rf in out
mkdir in
mkdir out

echo "generating seed files..."
cp ../tests/specs/* in/
grep -e 'let spec = ".*";$' -r ../frontend -h \
  | uniq \
  | sed -e 's/.*= "\(.*\)";/\1/' -e 's/\\n/ /g' \
  | tr '\n' '\0' \
  | xargs -I{} -0 -n1 \
    sh -c \
      'echo "{}" > in/$(echo "{}"|sha1sum -|sed -e "s/^\(.\{8\}\).*/\1/")'

pushd .. >/dev/null
echo "building rtlola-analyze..."
cargo build --bin rtlola-analyze --quiet
echo "test files with rtlola-analyze..."
for file in afl-fuzz-frontend/in/*
do
    echo | ./target/debug/rtlola-analyze "$file" ir 2>"$$" >/dev/null
    n_panics=$(grep -e 'panic' "$$"|wc -l)
    if [ $n_panics -gt 0 ]; then
        rm "$file"
    fi
done
rm "$$"
popd >/dev/null
