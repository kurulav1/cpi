#pragma once

// Common project-wide utilities and macros shared across all engine components.
// Currently provides the LLAMA_ENGINE_THROW macro for uniform error reporting.

#include <stdexcept>
#include <string>

// Throws a std::runtime_error with a "[llama_engine]" prefix prepended to msg.
// Use this macro throughout the engine instead of throwing directly so that
// callers can reliably identify exceptions originating from this library.
#define LLAMA_ENGINE_THROW(msg) throw std::runtime_error(std::string("[llama_engine] ") + (msg))
