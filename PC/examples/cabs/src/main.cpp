#include <signal.h>
#include <memory>

#include <cabs/cabs.h>
#include <cabs/application.h>
#include <cabs/screen.h>
#include <cabs/widget.h>

#include "sample_screen.h"

static void signal_handler(int sig)
{
	(void) sig;

	endwin();

	exit(0);
}

int main(void)
{
	signal(SIGINT, signal_handler);

	application << *new SampleScreen();

	application.run();

	return 0;
}
