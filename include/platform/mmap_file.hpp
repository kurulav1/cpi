#pragma once

// Cross-platform read-only memory-mapped file abstraction.
//
// MMapFile maps a file into the process address space without copying it into
// heap memory. This is the primary mechanism for loading large model weight
// files: the OS page cache serves weight data on demand and the process never
// allocates a contiguous RAM buffer for the entire file.
//
// On Windows the implementation uses CreateFile / CreateFileMapping /
// MapViewOfFile. On POSIX systems it uses open(2) / fstat(2) / mmap(2).
//
// The class is move-only (copy is deleted) so ownership of the mapping is
// unambiguous. Moving transfers the OS handles to the new instance and leaves
// the source in the default (invalid) state.

#include <cstddef>
#include <string>

namespace platform {

// Read-only memory-mapped view of a file on disk.
// Designed for large model weights to avoid loading entire blobs into RAM.
class MMapFile {
 public:
  MMapFile() = default;

  // Opens and maps the file at the given path immediately.
  // Throws std::runtime_error on any OS failure.
  explicit MMapFile(const std::string& path);

  // Unmaps the view and closes all OS handles.
  ~MMapFile();

  // Non-copyable: two instances cannot share ownership of the same OS mapping.
  MMapFile(const MMapFile&) = delete;
  MMapFile& operator=(const MMapFile&) = delete;

  // Move construction transfers OS handles; source becomes invalid.
  MMapFile(MMapFile&& other) noexcept;
  MMapFile& operator=(MMapFile&& other) noexcept;

  // Opens the file at path and maps it. Closes any existing mapping first.
  // Throws std::runtime_error on failure.
  void open(const std::string& path);

  // Unmaps and closes the file. Safe to call on an already-closed instance.
  void close();

  // Returns a pointer to the start of the mapped region, or nullptr if closed.
  [[nodiscard]] const std::byte* data() const { return data_; }

  // Returns the size of the mapped region in bytes.
  [[nodiscard]] std::size_t size() const { return size_; }

  // Returns true when the mapping is open and non-empty.
  [[nodiscard]] bool valid() const { return data_ != nullptr && size_ > 0; }

  // Hint to the OS to prefetch the entire mapped region into the page cache.
  // Non-blocking: the OS services the hint asynchronously. Best called immediately
  // after open() so disk I/O overlaps with GPU/CPU setup work.
  // No-op if the mapping is not valid.
  void prefetch() const;

 private:
  const std::byte* data_ = nullptr;  // base address of the mapped view
  std::size_t size_ = 0;             // byte length of the mapped view

#ifdef _WIN32
  // Windows requires two HANDLEs: one for the file, one for the mapping object.
  void* file_handle_ = nullptr;
  void* mapping_handle_ = nullptr;
#else
  int fd_ = -1;  // POSIX file descriptor; closed after mmap succeeds
#endif
};

}  // namespace platform
