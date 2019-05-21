#include <signal.h>
#include <memory>

#include <cabs/cabs.h>
#include <cabs/screen.h>

static void signal_handler(int sig)
{
	(void) sig;

	endwin();

	exit(0);
}

int main(void)
{
	signal(SIGINT, signal_handler);

	Screen screen;

	Widget sample_widget(20, 10, 21, 10, "SAMPLE WIDGET", true, true);
	screen << sample_widget;

	screen.redraw();

	getch();

	return 0;
}
