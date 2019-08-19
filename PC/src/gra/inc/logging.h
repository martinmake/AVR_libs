#ifndef _LOGGING_H_
#define _LOGGING_H_

#include <logger/logger.h>

#define LOG
#ifdef LOG
namespace Gra
{
	extern Logger::Console* logger;
}

#define TRACE(...) LOG_WRAPPER_TRACE((*Gra::logger), __VA_ARGS__)
#define INFO( ...) LOG_WRAPPER_INFO( (*Gra::logger), __VA_ARGS__)
#define WARN( ...) LOG_WRAPPER_WARN( (*Gra::logger), __VA_ARGS__)
#define ERROR(...) LOG_WRAPPER_ERROR((*Gra::logger), __VA_ARGS__)
#else
#define TRACE(...) { }
#define INFO( ...) { }
#define WARN( ...) { }
#define ERROR(...) { }
#endif

#endif
