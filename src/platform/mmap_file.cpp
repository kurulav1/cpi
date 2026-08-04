#include "platform/mmap_file.hpp"

#include <cstdlib>
#include <filesystem>

#include "common.hpp"

#ifdef _WIN32
#include <Windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace platform {

MMapFile::MMapFile(const std::string& path) {
  open(path);
}

MMapFile::~MMapFile() {
  close();
}

MMapFile::MMapFile(MMapFile&& other) noexcept {
  *this = std::move(other);
}

MMapFile& MMapFile::operator=(MMapFile&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  close();
  data_ = other.data_;
  size_ = other.size_;
#ifdef _WIN32
  file_handle_ = other.file_handle_;
  mapping_handle_ = other.mapping_handle_;
  other.file_handle_ = nullptr;
  other.mapping_handle_ = nullptr;
#else
  fd_ = other.fd_;
  other.fd_ = -1;
#endif
  other.data_ = nullptr;
  other.size_ = 0;
  return *this;
}

void MMapFile::open(const std::string& path) {
  close();

  if (!std::filesystem::exists(path)) {
    CPI_THROW("weights file not found: " + path);
  }

#ifdef _WIN32
  // FILE_FLAG_SEQUENTIAL_SCAN tells the cache manager to do aggressive
  // read-ahead, which matters because we stream the entire weight file to the
  // GPU in one pass right after opening it.
  file_handle_ = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                             FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
  if (file_handle_ == INVALID_HANDLE_VALUE) {
    file_handle_ = nullptr;
    CPI_THROW("CreateFileA failed for: " + path);
  }

  LARGE_INTEGER li;
  if (!GetFileSizeEx(static_cast<HANDLE>(file_handle_), &li)) {
    close();
    CPI_THROW("GetFileSizeEx failed");
  }
  size_ = static_cast<std::size_t>(li.QuadPart);

  mapping_handle_ =
      CreateFileMappingA(static_cast<HANDLE>(file_handle_), nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (!mapping_handle_) {
    close();
    CPI_THROW("CreateFileMappingA failed");
  }

  data_ = static_cast<const std::byte*>(
      MapViewOfFile(static_cast<HANDLE>(mapping_handle_), FILE_MAP_READ, 0, 0, 0));
  if (!data_) {
    close();
    CPI_THROW("MapViewOfFile failed");
  }
#else
  fd_ = ::open(path.c_str(), O_RDONLY);
  if (fd_ < 0) {
    CPI_THROW("open failed for: " + path);
  }

  struct stat st;
  if (fstat(fd_, &st) != 0) {
    close();
    CPI_THROW("fstat failed");
  }

  size_ = static_cast<std::size_t>(st.st_size);
  data_ = static_cast<const std::byte*>(mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0));
  if (data_ == MAP_FAILED) {
    data_ = nullptr;
    close();
    CPI_THROW("mmap failed");
  }
#endif
}

// Available physical RAM in bytes (0 if it can't be determined).
static std::size_t available_physical_ram() {
#ifdef _WIN32
  MEMORYSTATUSEX ms;
  ms.dwLength = sizeof(ms);
  return GlobalMemoryStatusEx(&ms) ? static_cast<std::size_t>(ms.ullAvailPhys) : 0;
#else
  const long pages = sysconf(_SC_AVPHYS_PAGES);
  const long page_size = sysconf(_SC_PAGESIZE);
  return (pages > 0 && page_size > 0) ? static_cast<std::size_t>(pages) * page_size : 0;
#endif
}

void MMapFile::prefetch() const {
  if (!valid()) {
    return;
  }
  // Skip the whole-file read-ahead when it would thrash. A model larger than free RAM cannot be
  // held in the page cache, so prefetching it just double-reads: the OS blocks in the prefetch
  // call, then its background read evicts pages before the loader reaches them and competes with
  // the loader's own mmap faults. Measured on the 32B int4 (23 GB, ~12 GB free): skipping this
  // ~halves cold start (44s -> 20s). Small models that fit in RAM still benefit from the prefetch.
  // CPI_NO_MMAP_PREFETCH=1 forces skip; CPI_FORCE_MMAP_PREFETCH=1 forces the read-ahead.
  if (std::getenv("CPI_NO_MMAP_PREFETCH") != nullptr) {
    return;
  }
  if (std::getenv("CPI_FORCE_MMAP_PREFETCH") == nullptr) {
    const std::size_t avail = available_physical_ram();
    if (avail > 0 && size_ > avail) {
      return;  // would not fit in the page cache -> prefetch would only double the disk traffic
    }
  }
#ifdef _WIN32
  // PrefetchVirtualMemory asynchronously reads the pages into the working set.
  // Available since Windows 8. Fails silently if the API is unavailable.
  WIN32_MEMORY_RANGE_ENTRY entry;
  entry.VirtualAddress = const_cast<std::byte*>(data_);
  entry.NumberOfBytes = size_;
  PrefetchVirtualMemory(GetCurrentProcess(), 1, &entry, 0);
#else
  // MADV_WILLNEED tells the kernel to read-ahead the mapped region
  // into the page cache. The call returns immediately.
  madvise(const_cast<std::byte*>(data_), size_, MADV_WILLNEED);
#endif
}

void MMapFile::close() {
#ifdef _WIN32
  if (data_) {
    UnmapViewOfFile(data_);
  }
  if (mapping_handle_) {
    CloseHandle(static_cast<HANDLE>(mapping_handle_));
    mapping_handle_ = nullptr;
  }
  if (file_handle_) {
    CloseHandle(static_cast<HANDLE>(file_handle_));
    file_handle_ = nullptr;
  }
#else
  if (data_ && size_ > 0) {
    munmap(const_cast<std::byte*>(data_), size_);
  }
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
#endif
  data_ = nullptr;
  size_ = 0;
}

}  // namespace platform