#ifndef _CABS_WIDGETS_BOX_H_
#define _CABS_WIDGETS_BOX_H_

#include "widget.h"

class Box : public Widget
{
	public:
		Box(int x, int y, int w, int h);
		~Box();
};

#endif
