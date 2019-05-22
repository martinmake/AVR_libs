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
		Widget(int x, int y, int w, int h);
		~Widget(void);

	public:
		void attatch_to_window(WINDOW* win);
		void draw(void);

	// GETTERS
	public:
		const std::string& label (void);
		bool               box   (void);
		bool               shadow(void);

	// SETTERS
	public:
		void label (const std::string& new_label );
		void box   (      bool         new_box   );
		void shadow(      bool         new_shadow);
};

// GETTERS
inline const std::string& Widget::label(void)
{
	return m_label;
}
inline bool Widget::box(void)
{
	return m_box;
}
inline bool Widget::shadow(void)
{
	return m_shadow;
}

// SETTERS
inline void Widget::label(const std::string& new_label)
{
	m_label = new_label;
}
inline void Widget::box(bool new_box)
{
	m_box = new_box;
}
inline void Widget::shadow(bool new_shadow)
{
	m_shadow = new_shadow;
}

#endif
