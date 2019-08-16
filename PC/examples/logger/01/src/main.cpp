#define CREATE_DEFAULT_LOGGER
#include <logger/logger.h>

#define TEXT "{0} is divisible by {1}"

int main(void)
{
	for (uint8_t i = 1; i; i++)
	{
		if (i % 2 == 0)
			LOG_INFO(TEXT, i, 2);
		if (i % 3 == 0)
			LOG_TRACE(TEXT, i, 3);
		if (i % 4 == 0)
			LOG_WARN(TEXT, i, 4);
		if (i % 5 == 0)
			LOG_ERROR(TEXT, i, 5);
	}
}
