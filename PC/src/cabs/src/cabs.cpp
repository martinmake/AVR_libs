#include "cabs.h"

namespace Cabs
{
	namespace Position
	{
		int LEFT   =  0;
		int TOP    =  0;
		int CENTER = -1;
		int RIGHT  = -2;
		int BOTTOM = -3;

		int translate_x(const WINDOW* win, int x, int w)
		{
			if (x == CENTER)
				return (getmaxx(win) - w) / 2;
			if (x == RIGHT)
				return getmaxx(win) - w;

			return x;
		}

		int translate_y(const WINDOW* win, int y, int h)
		{
			if (y == CENTER)
				return (getmaxy(win) - h) / 2;
			if (y == BOTTOM)
				return getmaxy(win) - h;

			return y;
		}
	}
}
