#include "sample_screen.h"

SampleScreen::SampleScreen(void)
{
	#include "dsn/sample.cpp"

	// construction may continue...
}

SampleScreen::~SampleScreen(void)
{
}

void SampleScreen::handle_key(int key)
{
	if (key == 'q')
		application.exit(0);
	else
		draw();
}
