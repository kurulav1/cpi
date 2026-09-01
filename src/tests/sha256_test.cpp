// SHA-256 against the published vectors.
//
// A hand-rolled digest that is subtly wrong is worse than no digest: it would
// produce stable, plausible hex that agrees with itself and with nothing else in
// the world, so a reader comparing our "sha256" against theirs would see a
// mismatch and go looking for a determinism bug that does not exist. These are the
// standard FIPS 180-4 / NIST test vectors, so agreement here means the value we
// print is the value everyone else calls SHA-256.

#include "util/sha256.hpp"

#include <cstdio>
#include <string>

namespace {

int failures = 0;

void expect(const std::string& label, const std::string& got, const std::string& want) {
  if (got == want) {
    std::printf("  ok   %-28s %s\n", label.c_str(), got.substr(0, 16).c_str());
  } else {
    std::printf("  FAIL %-28s\n       got  %s\n       want %s\n", label.c_str(), got.c_str(),
                want.c_str());
    ++failures;
  }
}

}  // namespace

int main() {
  using cpi::util::Sha256;
  using cpi::util::sha256_hex;

  std::printf("sha256 vectors\n");

  expect("empty string", sha256_hex(""),
         "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  expect("abc", sha256_hex("abc"),
         "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  expect("448-bit message", sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
         "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  expect("896-bit message",
         sha256_hex("abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnop"
                    "jklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu"),
         "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1");

  // A million 'a', fed in awkward chunk sizes. The point is the block boundary
  // handling: a digest that is right only when the input arrives in 64-byte pieces
  // would pass every test above and fail on a real file read.
  {
    Sha256 h;
    const std::string chunk(1000, 'a');
    for (int i = 0; i < 1000; ++i) {
      h.update(chunk);
    }
    expect("one million 'a'", h.hex(),
           "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
  }

  // Length exactly at the padding boundaries, where the "does the length fit in
  // this block" branch flips. 55 and 56 bytes are the two sides of it, and 64 is a
  // whole block with nothing left over.
  expect("55 bytes", sha256_hex(std::string(55, 'x')),
         "d5e285683cd4efc02d021a5c62014694958901005d6f71e89e0989fac77e4072");
  expect("56 bytes", sha256_hex(std::string(56, 'x')),
         "04c26261370ee7541549d16dee320c723e3fd14671e66a099afe0a377c16888e");
  expect("63 bytes", sha256_hex(std::string(63, 'x')),
         "75220b47218278e656f2013bb8f0c455a25eaf01e86c64924e9d48d89776d6f2");
  expect("64 bytes", sha256_hex(std::string(64, 'x')),
         "7ce100971f64e7001e8fe5a51973ecdfe1ced42befe7ee8d5fd6219506b5393c");

  if (failures != 0) {
    std::printf("sha256_test: %d FAILED\n", failures);
    return 1;
  }
  std::printf("sha256_test: all vectors match\n");
  return 0;
}
