#ifndef _CABS_CABS_H_
#define _CABS_CABS_H_

#include <ncurses.h>

namespace Cabs
{
	namespace Position
	{
		extern int LEFT;
		extern int TOP;
		extern int CENTER;
		extern int RIGHT;
		extern int BOTTOM;

		extern int translate_x(const WINDOW* win, int x, int w);
		extern int translate_y(const WINDOW* win, int y, int h);
	}
}

#endif
