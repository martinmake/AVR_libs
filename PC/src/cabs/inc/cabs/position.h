#ifndef _CABS_POSITION_H_
#define _CABS_POSITION_H_

#include <ncurses.h>

#include "cabs.h"

class Position
{

	private:
		int m_x = 0;
		int m_y = 0;

	public:
		Position(int initial_x, int initial_y);
		Position(void);
		~Position(void);

	// GETTERS
	public:
		int x(void)                 const;
		int x(const WINDOW*, int w) const;
		int y(void)                 const;
		int y(const WINDOW*, int h) const;

	// SETTERS
	public:
		void x(int new_x);
		void y(int new_y);
};

// GETTERS
inline int Position::x(void) const
{
	return m_x;
}
inline int Position::x(const WINDOW* win, int w) const
{
	using namespace Cabs::Positions;

	if (m_x == CENTER)
		return (getmaxx(win) - w) / 2;
	if (m_x == RIGHT)
		return getmaxx(win) - w - 3;

	return m_x;
}
inline int Position::y(void) const
{
	return m_y;
}
inline int Position::y(const WINDOW* win, int h) const
{
	using namespace Cabs::Positions;

	if (m_y == CENTER)
		return (getmaxy(win) - h) / 2;
	if (m_y == BOTTOM)
		return getmaxy(win) - h - 3;

	return m_y;
}

// SETTERS
inline void Position::x(int new_x)
{
	m_x = new_x;
}
inline void Position::y(int new_y)
{
	m_y = new_y;
}

#endif
