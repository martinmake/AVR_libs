#include <logger/logger.h>

#define LOG

#ifdef LOG
Logger::Console example_logger("EXAMPLE");
#define EXAMPLE_TRACE(...) LOG_WRAPPER_TRACE(example_logger, __VA_ARGS__)
#define EXAMPLE_INFO( ...) LOG_WRAPPER_INFO( example_logger, __VA_ARGS__)
#define EXAMPLE_WARN( ...) LOG_WRAPPER_WARN( example_logger, __VA_ARGS__)
#define EXAMPLE_ERROR(...) LOG_WRAPPER_ERROR(example_logger, __VA_ARGS__)
#else
#define EXAMPLE_TRACE(...) { }
#define EXAMPLE_INFO( ...) { }
#define EXAMPLE_WARN( ...) { }
#define EXAMPLE_ERROR(...) { }
#endif

#define TEXT "{0} is divisible by {1}"

int main(void)
{
	for (uint8_t i = 1; i; i++)
	{
		if (i % 1 == 0)
			EXAMPLE_INFO(TEXT, i, 1);
		if (i % 3 == 0)
			EXAMPLE_TRACE(TEXT, i, 3);
		if (i % 10 == 0)
			EXAMPLE_WARN(TEXT, i, 10);
		if (i % 50 == 0)
			EXAMPLE_ERROR(TEXT, i, 50);
	}
}
