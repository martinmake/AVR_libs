#ifndef _CABS_WIDGET_H_
#define _CABS_WIDGET_H_

#include <ncurses.h>
#include <string>
#include <list>

#include "cabs.h"

class Widget
{
	protected:
		WINDOW*     m_win;
		WINDOW*     m_win_shadow;
		std::string m_label;
		bool        m_box;
		bool        m_shadow;
		int         m_x;
		int	    m_y;
		int	    m_w;
		int	    m_h;

	public:
		Widget(int x, int y, int w, int h, const std::string& label = "", bool border = true, bool shadow = false);
		~Widget(void);

	public:
		void attatch_to_window(WINDOW* win);
		void draw(void);
};

#endif
