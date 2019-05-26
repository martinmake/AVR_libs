#include <ncurses.h>

#include "cabs/cabs.h"

namespace Cabs
{
	namespace Positions
	{
		const int LEFT   =  0;
		const int TOP    =  0;
		const int CENTER = -1;
		const int RIGHT  = -2;
		const int BOTTOM = -3;
	}

	namespace Colors
	{
		int pair = 0;

		const int NO_COLOR        = COLOR_PAIR(pair++);

		const int BLACK_BLACK     = COLOR_PAIR(pair++);
		const int RED_BLACK       = COLOR_PAIR(pair++);
		const int GREEN_BLACK     = COLOR_PAIR(pair++);
		const int YELLOW_BLACK    = COLOR_PAIR(pair++);
		const int BLUE_BLACK      = COLOR_PAIR(pair++);
		const int MAGENTA_BLACK   = COLOR_PAIR(pair++);
		const int CYAN_BLACK      = COLOR_PAIR(pair++);
		const int WHITE_BLACK     = COLOR_PAIR(pair++);

		const int BLACK_RED       = COLOR_PAIR(pair++);
		const int RED_RED         = COLOR_PAIR(pair++);
		const int GREEN_RED       = COLOR_PAIR(pair++);
		const int YELLOW_RED      = COLOR_PAIR(pair++);
		const int BLUE_RED        = COLOR_PAIR(pair++);
		const int MAGENTA_RED     = COLOR_PAIR(pair++);
		const int CYAN_RED        = COLOR_PAIR(pair++);
		const int WHITE_RED       = COLOR_PAIR(pair++);

		const int BLACK_GREEN     = COLOR_PAIR(pair++);
		const int RED_GREEN       = COLOR_PAIR(pair++);
		const int GREEN_GREEN     = COLOR_PAIR(pair++);
		const int YELLOW_GREEN    = COLOR_PAIR(pair++);
		const int BLUE_GREEN      = COLOR_PAIR(pair++);
		const int MAGENTA_GREEN   = COLOR_PAIR(pair++);
		const int CYAN_GREEN      = COLOR_PAIR(pair++);
		const int WHITE_GREEN     = COLOR_PAIR(pair++);

		const int BLACK_YELLOW    = COLOR_PAIR(pair++);
		const int RED_YELLOW      = COLOR_PAIR(pair++);
		const int GREEN_YELLOW    = COLOR_PAIR(pair++);
		const int YELLOW_YELLOW   = COLOR_PAIR(pair++);
		const int BLUE_YELLOW     = COLOR_PAIR(pair++);
		const int MAGENTA_YELLOW  = COLOR_PAIR(pair++);
		const int CYAN_YELLOW     = COLOR_PAIR(pair++);
		const int WHITE_YELLOW    = COLOR_PAIR(pair++);

		const int BLACK_BLUE      = COLOR_PAIR(pair++);
		const int RED_BLUE        = COLOR_PAIR(pair++);
		const int GREEN_BLUE      = COLOR_PAIR(pair++);
		const int YELLOW_BLUE     = COLOR_PAIR(pair++);
		const int BLUE_BLUE       = COLOR_PAIR(pair++);
		const int MAGENTA_BLUE    = COLOR_PAIR(pair++);
		const int CYAN_BLUE       = COLOR_PAIR(pair++);
		const int WHITE_BLUE      = COLOR_PAIR(pair++);

		const int BLACK_MAGENTA   = COLOR_PAIR(pair++);
		const int RED_MAGENTA     = COLOR_PAIR(pair++);
		const int GREEN_MAGENTA   = COLOR_PAIR(pair++);
		const int YELLOW_MAGENTA  = COLOR_PAIR(pair++);
		const int BLUE_MAGENTA    = COLOR_PAIR(pair++);
		const int MAGENTA_MAGENTA = COLOR_PAIR(pair++);
		const int CYAN_MAGENTA    = COLOR_PAIR(pair++);
		const int WHITE_MAGENTA   = COLOR_PAIR(pair++);

		const int BLACK_CYAN      = COLOR_PAIR(pair++);
		const int RED_CYAN        = COLOR_PAIR(pair++);
		const int GREEN_CYAN      = COLOR_PAIR(pair++);
		const int YELLOW_CYAN     = COLOR_PAIR(pair++);
		const int BLUE_CYAN       = COLOR_PAIR(pair++);
		const int MAGENTA_CYAN    = COLOR_PAIR(pair++);
		const int CYAN_CYAN       = COLOR_PAIR(pair++);
		const int WHITE_CYAN      = COLOR_PAIR(pair++);

		const int BLACK_WHITE     = COLOR_PAIR(pair++);
		const int RED_WHITE       = COLOR_PAIR(pair++);
		const int GREEN_WHITE     = COLOR_PAIR(pair++);
		const int YELLOW_WHITE    = COLOR_PAIR(pair++);
		const int BLUE_WHITE      = COLOR_PAIR(pair++);
		const int MAGENTA_WHITE   = COLOR_PAIR(pair++);
		const int CYAN_WHITE      = COLOR_PAIR(pair++);
		const int WHITE_WHITE     = COLOR_PAIR(pair++);
	}

	int parse_tag(const std::string& tag)
	{
		NO_COLOR

		BLACK_BLACK
		RED_BLACK
		GREEN_BLACK
		YELLOW_BLACK
		BLUE_BLACK
		MAGENTA_BLACK
		CYAN_BLACK
		WHITE_BLACK

		BLACK_RED
		RED_RED
		GREEN_RED
		YELLOW_RED
		BLUE_RED
		MAGENTA_RED
		CYAN_RED
		WHITE_RED

		BLACK_GREEN
		RED_GREEN
		GREEN_GREEN
		YELLOW_GREEN
		BLUE_GREEN
		MAGENTA_GREEN
		CYAN_GREEN
		WHITE_GREEN

		BLACK_YELLOW
		RED_YELLOW
		GREEN_YELLOW
		YELLOW_YELLOW
		BLUE_YELLOW
		MAGENTA_YELLOW
		CYAN_YELLOW
		WHITE_YELLOW

		BLACK_BLUE
		RED_BLUE
		GREEN_BLUE
		YELLOW_BLUE
		BLUE_BLUE
		MAGENTA_BLUE
		CYAN_BLUE
		WHITE_BLUE

		BLACK_MAGENTA
		RED_MAGENTA
		GREEN_MAGENTA
		YELLOW_MAGENTA
		BLUE_MAGENTA
		MAGENTA_MAGENTA
		CYAN_MAGENTA
		WHITE_MAGENTA

		BLACK_CYAN
		RED_CYAN
		GREEN_CYAN
		YELLOW_CYAN
		BLUE_CYAN
		MAGENTA_CYAN
		CYAN_CYAN
		WHITE_CYAN

		BLACK_WHITE
		RED_WHITE
		GREEN_WHITE
		YELLOW_WHITE
		BLUE_WHITE
		MAGENTA_WHITE
		CYAN_WHITE
		WHITE_WHITE
	}

	bool move = false;
}
