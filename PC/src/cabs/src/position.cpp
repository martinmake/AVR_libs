#include "cabs/position.h"

Position::Position(int initial_x,int initial_y)
	: m_x(initial_x), m_y(initial_y)
{
}

Position::Position(void)
{
}

Position::~Position(void)
{
}

int Position::translate_x(const WINDOW* win, int x, int w)
{
	using namespace Cabs::Positions;

	if (x == CENTER)
		return (getmaxx(win) - w) / 2;
	if (x == RIGHT)
		return getmaxx(win) - w - 1;

	return x;
}

int Position::translate_y(const WINDOW* win, int y, int h)
{
	using namespace Cabs::Positions;

	if (y == CENTER)
		return (getmaxy(win) - h) / 2;
	if (y == BOTTOM)
		return getmaxy(win) - h - 1;

	return y;
}
