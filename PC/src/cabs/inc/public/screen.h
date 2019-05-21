#ifndef _CABS_SCREEN_H_
#define _CABS_SCREEN_H_

#include <ncurses.h>
#include <string>
#include <list>

#include "widget.h"

class Screen
{
	private:
		std::list<Widget> m_widgets;
		std::list<bool>   m_widget_states;

	protected:
		WINDOW* m_win;
		int     m_x;
		int	m_y;
		int	m_w;
		int	m_h;

	public:
		Screen(int x, int y, int w, int h);
		Screen(void);
		~Screen(void);

	public:
		void redraw(void);
		Screen& operator<<(Widget& widget);
};

#endif
