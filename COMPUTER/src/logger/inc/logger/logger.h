#ifndef _LOGGER_LOGGER_H_
#define _LOGGER_LOGGER_H_

#include "logger/loggers/all.h"

#define LOG_WRAPPER_TRACE(logger, ...) logger.underlying_logger()->trace(__VA_ARGS__);
#define LOG_WRAPPER_INFO( logger, ...) logger.underlying_logger()->info (__VA_ARGS__);
#define LOG_WRAPPER_WARN( logger, ...) logger.underlying_logger()->warn (__VA_ARGS__);
#define LOG_WRAPPER_ERROR(logger, ...) logger.underlying_logger()->error(__VA_ARGS__);

namespace Logger
{
#ifdef CREATE_DEFAULT_LOGGER
	extern Logger::Console default_logger;

	#define LOG_TRACE(...) LOG_WRAPPER_TRACE(Logger::default_logger, __VA_ARGS__)
	#define LOG_INFO( ...) LOG_WRAPPER_INFO( Logger::default_logger, __VA_ARGS__)
	#define LOG_WARN( ...) LOG_WRAPPER_WARN( Logger::default_logger, __VA_ARGS__)
	#define LOG_ERROR(...) LOG_WRAPPER_ERROR(Logger::default_logger, __VA_ARGS__)
#endif
}

#endif
