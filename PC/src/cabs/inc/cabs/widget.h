#ifndef _CABS_WIDGET_H_
#define _CABS_WIDGET_H_

#include <string>

#include "cabs/position.h"
#include "cabs/size.h"

#undef shadow_attr
#undef border_attr
#undef is_bordered

class Widget
{
	protected:
		WINDOW*     m_win_border;
		WINDOW*     m_win_shadow;
		std::string m_label;
		Position    m_position;
		Size	    m_size;
		bool        m_is_bordered = false;
		bool        m_is_shadowed = false;
		bool        m_is_visible  = false;
		bool        m_is_selected = false;
		int         m_label_attr  = 0;
		int         m_border_attr = 0;
		int         m_shadow_attr = 0;

	public:
		Widget(void);
		virtual ~Widget(void);

	public:
		void attatch_to_window(WINDOW* win);
		virtual void draw(void) const;

	// HANDLERS
	public:
		virtual void handle_key(int key);

	// GETTERS
	public:
		const Position&    position   (void) const;
		const Size&        size       (void) const;
		const std::string& label      (void) const;
		      bool         is_bordered(void) const;
		      bool         is_shadowed(void) const;
		      bool         is_visible (void) const;
		      bool         is_selected(void) const;
		      int          label_attr (void) const;
		      int          border_attr(void) const;
		      int          shadow_attr(void) const;

	// SETTERS
	public:
		void position   (const Position&    new_position   );
		void size       (const Size&        new_size       );
		void label      (const std::string& new_label      );
		void is_bordered(      bool         new_is_bordered);
		void is_shadowed(      bool         new_is_shadowed);
		void is_visible (      bool         new_is_visible );
		void is_selected(      bool         new_is_selected);
		void label_attr (      int          new_label_attr );
		void border_attr(      int          new_border_attr);
		void shadow_attr(      int          new_shadow_attr);
};

// GETTERS
inline const Position& Widget::position(void) const
{
	return m_position;
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
inline int Widget::label_attr(void) const
{
	return m_label_attr;
}
inline int Widget::border_attr(void) const
{
	return m_border_attr;
}
inline int Widget::shadow_attr(void) const
{
	return m_shadow_attr;
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
inline void Widget::label_attr(int new_label_attr)
{
	m_label_attr = new_label_attr;
}
inline void Widget::border_attr(int new_border_attr)
{
	m_border_attr = new_border_attr;
}
inline void Widget::shadow_attr(int new_shadow_attr)
{
	m_shadow_attr = new_shadow_attr;
}

#endif
