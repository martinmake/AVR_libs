#include "logger/loggers/base.h"

namespace Logger
{
	Base::Base(void)
	{
	}
	Base::Base(const std::string& initial_name)
		: m_name(initial_name)
	{
	}

	Base::~Base(void)
	{
	}

	void Base::init(void)
	{
		m_underlying_logger->set_level(spdlog::level::trace);
	}

	void Base::copy(const Base& other)
	{
		(void) other;
	}
	void Base::move(Base&& other)
	{
		(void) other;
	}
}
