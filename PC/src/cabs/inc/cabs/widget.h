#ifndef _CABS_WIDGET_H_
#define _CABS_WIDGET_H_

#include <string>
#include <ncurses.h>

#include "cabs/position.h"
#include "cabs/size.h"

#undef shadow_attr
#undef border_attr
#undef is_bordered

class Widget
{
	protected:
		WINDOW*     m_border_win;
		WINDOW*     m_shadow_win;
		std::string m_label;
		Position    m_position;
		Size	    m_size;
		bool        m_is_bordered = false;
		bool        m_is_shadowed = false;
		bool        m_is_visible  = false;
		bool        m_is_selected = false;

	public:
		Widget(void);
		virtual ~Widget(void);

	public:
		void attatch_to_window(WINDOW* win);
		virtual void draw(void) const;

	protected:
		virtual void draw_inside(void) const;
		virtual void clear_inside(void) const;

	// HANDLERS
	public:
		virtual void handle_key(int key);

	// GETTERS
	public:
		virtual const Position&    position    (       void      ) const;
		virtual       Position     position    (const WINDOW* win) const;
		virtual const Size&        size        (       void      ) const;
		virtual const std::string& label       (       void      ) const;
		virtual       bool         is_bordered (       void      ) const;
		virtual       bool         is_shadowed (       void      ) const;
		virtual       bool         is_visible  (       void      ) const;
		virtual       bool         is_selected (       void      ) const;

	// SETTERS
	public:
		virtual void position   (const Position&    new_position   );
		virtual void size       (const Size&        new_size       );
		virtual void label      (const std::string& new_label      );
		virtual void is_bordered(      bool         new_is_bordered);
		virtual void is_shadowed(      bool         new_is_shadowed);
		virtual void is_visible (      bool         new_is_visible );
		virtual void is_selected(      bool         new_is_selected);
};

// GETTERS
inline const Position& Widget::position(void) const
{
	return m_position;
}
inline Position Widget::position(const WINDOW* win) const
{
	return { m_position.x(win, m_size.w()), m_position.y(win, m_size.h()) };
}
inline const Size& Widget::size(void) const
{
	return m_size;
}
inline const std::string& Widget::label(void) const
{
	return m_label;
}
inline bool Widget::is_bordered(void) const
{
	return m_is_bordered;
}
inline bool Widget::is_shadowed(void) const
{
	return m_is_shadowed;
}
inline bool Widget::is_visible(void) const
{
	return m_is_visible;
}
inline bool Widget::is_selected(void) const
{
	return m_is_selected;
}

// SETTERS
inline void Widget::position(const Position& new_position)
{
	m_position = new_position;
}
inline void Widget::size(const Size& new_size)
{
	m_size = new_size;
}
inline void Widget::label(const std::string& new_label)
{
	m_label = new_label;
}
inline void Widget::is_bordered(bool new_is_bordered)
{
	m_is_bordered = new_is_bordered;
}
inline void Widget::is_shadowed(bool new_is_shadowed)
{
	m_is_shadowed = new_is_shadowed;
}
inline void Widget::is_visible(bool new_is_visible)
{
	m_is_visible = new_is_visible;
}
inline void Widget::is_selected(bool new_is_selected)
{
	m_is_selected = new_is_selected;
}

#endif
