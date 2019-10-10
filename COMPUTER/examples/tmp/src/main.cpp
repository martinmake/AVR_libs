#include <unistd.h>

#define CREATE_DEFAULT_LOGGER
#include <logger/logger.h>

int main(void)
{
	LOG_INFO("UF");
	sleep(1);
}
