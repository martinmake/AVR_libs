#ifndef _EXAMPLES_CABS_SAMPLE_SCREEN_H_
#define _EXAMPLES_CABS_SAMPLE_SCREEN_H_

#include <cabs/screen.h>
#include <cabs/widgets/text_box.h>

class SampleScreen : public Screen
{
	// WIDGETS
	public:
		#include "dsn/sample.h"

	public:
		SampleScreen(void);
		~SampleScreen(void);

	// HANDLERS
	public:
		void handle_key(int key) override;
};

#endif
