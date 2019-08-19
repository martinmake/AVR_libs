#include <spdlog/sinks/stdout_color_sinks.h>

#include "logger/loggers/console.h"

namespace Logger
{
	Console::Console(void)
	{
	}
	Console::Console(const std::string& initial_name)
		: Base(initial_name)
	{
		m_underlying_logger = spdlog::stdout_color_mt(m_name);
		init();
	}

	Console::~Console(void)
	{
	}

	void Console::copy(const Console& other)
	{
		Logger::Base::copy(other);
	}
	void Console::move(Console&& other)
	{
		Logger::Base::move(std::move(other));
	}
}
