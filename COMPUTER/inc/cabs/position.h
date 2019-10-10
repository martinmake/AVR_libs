#ifndef _CABS_POSITION_H_
#define _CABS_POSITION_H_

#include <ncurses.h>
#include <math.h>

#include "cabs.h"

class Position
{
	private:
		const WINDOW* m_parent_win = nullptr;
		float m_percentual_x = 0.0;
		float m_percentual_y = 0.0;

	public:
		Position(float initial_percentual_x, float initial_percentual_y);
		Position(void);
		~Position(void);

	// GETTERS
	public:
		int x(void) const;
		int y(void) const;

	// SETTERS
	public:
		void x(float new_percentual_x);
		void y(float new_percentual_y);
		void parent_win(const WINDOW* new_parent_win);
};

// GETTERS
inline int Position::x(void) const
{
	return round(getmaxx(m_parent_win) * m_percentual_x) + 1;
}
inline int Position::y(void) const
{
	return round(getmaxy(m_parent_win) * m_percentual_y) + 1;
}

// SETTERS
inline void Position::x(float new_percentual_x)
{
	m_percentual_x = new_percentual_x;
}
inline void Position::y(float new_percentual_y)
{
	m_percentual_y = new_percentual_y;
}
inline void Position::parent_win(const WINDOW* new_parent_win)
{
	m_parent_win = new_parent_win;
}

#endif
