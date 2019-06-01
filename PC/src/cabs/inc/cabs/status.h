#ifndef _CABS_STATUS_H_
#define _CABS_STATUS_H_

#include <ncurses.h>

#include "cabs/cabs.h"

class Status
{

	private:
		WINDOW* m_win;
		Cabs::Mode m_mode = Cabs::Mode::NORMAL;
		int m_mode_attr = Cabs::Colors::BLACK_GREEN;

	public:
		Status(void);
		~Status(void);

	public:
		void attatch_to_window(WINDOW* win);
		void redraw(void);
		void draw(void);

	// GETTERS
	public:
		Cabs::Mode mode(void) const;

	// SETTERS
	public:
		void mode(Cabs::Mode new_mode);
};

// GETTERS
inline Cabs::Mode Status::mode(void) const
{
	return m_mode;
}

// SETTERS
inline void Status::mode(Cabs::Mode new_mode)
{
	m_mode = new_mode;
}

#endif
