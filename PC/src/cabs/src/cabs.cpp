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
		using namespace Colors;

		if      (tag == "NO_COLOR")        return NO_COLOR;

		else if (tag == "BLACK_BLACK")     return BLACK_BLACK;
		else if (tag == "RED_BLACK")       return RED_BLACK;
		else if (tag == "GREEN_BLACK")     return GREEN_BLACK;
		else if (tag == "YELLOW_BLACK")    return YELLOW_BLACK;
		else if (tag == "BLUE_BLACK")      return BLUE_BLACK;
		else if (tag == "MAGENTA_BLACK")   return MAGENTA_BLACK;
		else if (tag == "CYAN_BLACK")      return CYAN_BLACK;
		else if (tag == "WHITE_BLACK")     return WHITE_BLACK;

		else if (tag == "BLACK_RED")       return BLACK_RED;
		else if (tag == "RED_RED")         return RED_RED;
		else if (tag == "GREEN_RED")       return GREEN_RED;
		else if (tag == "YELLOW_RED")      return YELLOW_RED;
		else if (tag == "BLUE_RED")        return BLUE_RED;
		else if (tag == "MAGENTA_RED")     return MAGENTA_RED;
		else if (tag == "CYAN_RED")        return CYAN_RED;
		else if (tag == "WHITE_RED")       return WHITE_RED;

		else if (tag == "BLACK_GREEN")     return BLACK_GREEN;
		else if (tag == "RED_GREEN")       return RED_GREEN;
		else if (tag == "GREEN_GREEN")     return GREEN_GREEN;
		else if (tag == "YELLOW_GREEN")    return YELLOW_GREEN;
		else if (tag == "BLUE_GREEN")      return BLUE_GREEN;
		else if (tag == "MAGENTA_GREEN")   return MAGENTA_GREEN;
		else if (tag == "CYAN_GREEN")      return CYAN_GREEN;
		else if (tag == "WHITE_GREEN")     return WHITE_GREEN;

		else if (tag == "BLACK_YELLOW")    return BLACK_YELLOW;
		else if (tag == "RED_YELLOW")      return RED_YELLOW;
		else if (tag == "GREEN_YELLOW")    return GREEN_YELLOW;
		else if (tag == "YELLOW_YELLOW")   return YELLOW_YELLOW;
		else if (tag == "BLUE_YELLOW")     return BLUE_YELLOW;
		else if (tag == "MAGENTA_YELLOW")  return MAGENTA_YELLOW;
		else if (tag == "CYAN_YELLOW")     return CYAN_YELLOW;
		else if (tag == "WHITE_YELLOW")    return WHITE_YELLOW;

		else if (tag == "BLACK_BLUE")      return BLACK_BLUE;
		else if (tag == "RED_BLUE")        return RED_BLUE;
		else if (tag == "GREEN_BLUE")      return GREEN_BLUE;
		else if (tag == "YELLOW_BLUE")     return YELLOW_BLUE;
		else if (tag == "BLUE_BLUE")       return BLUE_BLUE;
		else if (tag == "MAGENTA_BLUE")    return MAGENTA_BLUE;
		else if (tag == "CYAN_BLUE")       return CYAN_BLUE;
		else if (tag == "WHITE_BLUE")      return WHITE_BLUE;

		else if (tag == "BLACK_MAGENTA")   return BLACK_MAGENTA;
		else if (tag == "RED_MAGENTA")     return RED_MAGENTA;
		else if (tag == "GREEN_MAGENTA")   return GREEN_MAGENTA;
		else if (tag == "YELLOW_MAGENTA")  return YELLOW_MAGENTA;
		else if (tag == "BLUE_MAGENTA")    return BLUE_MAGENTA;
		else if (tag == "MAGENTA_MAGENTA") return MAGENTA_MAGENTA;
		else if (tag == "CYAN_MAGENTA")    return CYAN_MAGENTA;
		else if (tag == "WHITE_MAGENTA")   return WHITE_MAGENTA;

		else if (tag == "BLACK_CYAN")      return BLACK_CYAN;
		else if (tag == "RED_CYAN")        return RED_CYAN;
		else if (tag == "GREEN_CYAN")      return GREEN_CYAN;
		else if (tag == "YELLOW_CYAN")     return YELLOW_CYAN;
		else if (tag == "BLUE_CYAN")       return BLUE_CYAN;
		else if (tag == "MAGENTA_CYAN")    return MAGENTA_CYAN;
		else if (tag == "CYAN_CYAN")       return CYAN_CYAN;
		else if (tag == "WHITE_CYAN")      return WHITE_CYAN;

		else if (tag == "BLACK_WHITE")     return BLACK_WHITE;
		else if (tag == "RED_WHITE")       return RED_WHITE;
		else if (tag == "GREEN_WHITE")     return GREEN_WHITE;
		else if (tag == "YELLOW_WHITE")    return YELLOW_WHITE;
		else if (tag == "BLUE_WHITE")      return BLUE_WHITE;
		else if (tag == "MAGENTA_WHITE")   return MAGENTA_WHITE;
		else if (tag == "CYAN_WHITE")      return CYAN_WHITE;
		else if (tag == "WHITE_WHITE")     return WHITE_WHITE;
		else                               return NO_COLOR;
	}

	bool move = false;
}
