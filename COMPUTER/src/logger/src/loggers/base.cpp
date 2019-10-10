#include <iostream>

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
		m_underlying_logger = other.m_underlying_logger;
		m_name              = other.m_name;
	}
	void Base::move(Base&& other)
	{
		m_underlying_logger = std::move(other.m_underlying_logger);
		m_name              = other.m_name;
	}
}
