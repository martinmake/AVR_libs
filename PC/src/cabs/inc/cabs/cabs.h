#ifndef _CABS_CABS_H_
#define _CABS_CABS_H_

namespace Cabs
{
	namespace Positions
	{
		extern const int LEFT;
		extern const int TOP;
		extern const int CENTER;
		extern const int RIGHT;
		extern const int BOTTOM;
	}

	namespace Colors
	{
		extern const int NO_COLOR;

		extern const int BLACK_BLACK;
		extern const int RED_BLACK;
		extern const int GREEN_BLACK;
		extern const int YELLOW_BLACK;
		extern const int BLUE_BLACK;
		extern const int MAGENTA_BLACK;
		extern const int CYAN_BLACK;
		extern const int WHITE_BLACK;

		extern const int BLACK_RED;
		extern const int RED_RED;
		extern const int GREEN_RED;
		extern const int YELLOW_RED;
		extern const int BLUE_RED;
		extern const int MAGENTA_RED;
		extern const int CYAN_RED;
		extern const int WHITE_RED;

		extern const int BLACK_GREEN;
		extern const int RED_GREEN;
		extern const int GREEN_GREEN;
		extern const int YELLOW_GREEN;
		extern const int BLUE_GREEN;
		extern const int MAGENTA_GREEN;
		extern const int CYAN_GREEN;
		extern const int WHITE_GREEN;

		extern const int BLACK_YELLOW;
		extern const int RED_YELLOW;
		extern const int GREEN_YELLOW;
		extern const int YELLOW_YELLOW;
		extern const int BLUE_YELLOW;
		extern const int MAGENTA_YELLOW;
		extern const int CYAN_YELLOW;
		extern const int WHITE_YELLOW;

		extern const int BLACK_BLUE;
		extern const int RED_BLUE;
		extern const int GREEN_BLUE;
		extern const int YELLOW_BLUE;
		extern const int BLUE_BLUE;
		extern const int MAGENTA_BLUE;
		extern const int CYAN_BLUE;
		extern const int WHITE_BLUE;

		extern const int BLACK_MAGENTA;
		extern const int RED_MAGENTA;
		extern const int GREEN_MAGENTA;
		extern const int YELLOW_MAGENTA;
		extern const int BLUE_MAGENTA;
		extern const int MAGENTA_MAGENTA;
		extern const int CYAN_MAGENTA;
		extern const int WHITE_MAGENTA;

		extern const int BLACK_CYAN;
		extern const int RED_CYAN;
		extern const int GREEN_CYAN;
		extern const int YELLOW_CYAN;
		extern const int BLUE_CYAN;
		extern const int MAGENTA_CYAN;
		extern const int CYAN_CYAN;
		extern const int WHITE_CYAN;

		extern const int BLACK_WHITE;
		extern const int RED_WHITE;
		extern const int GREEN_WHITE;
		extern const int YELLOW_WHITE;
		extern const int BLUE_WHITE;
		extern const int MAGENTA_WHITE;
		extern const int CYAN_WHITE;
		extern const int WHITE_WHITE;
	}

	extern int parse_tag(const std::string& tag);

	extern bool move;
}

#endif
