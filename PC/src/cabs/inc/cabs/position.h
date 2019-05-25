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

	public:
		static int translate_x(const WINDOW* win, int x, int w);
		static int translate_y(const WINDOW* win, int y, int h);

	// GETTERS
	public:
		int x(void) const;
		int y(void) const;

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
inline int Position::y(void) const
{
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
