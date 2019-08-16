#include <spdlog/spdlog.h>

#include "initializer.h"

#define DEFAULT_LOG_PATTERN "%^[%T] %n: %v%$"

static Initializer initializer;

Initializer::Initializer(void)
{
	spdlog::set_pattern(DEFAULT_LOG_PATTERN);
}

Initializer::~Initializer(void)
{
}
