#ifndef _CABS_WIDGET_H_
#define _CABS_WIDGET_H_

#include <string>

#include "cabs/position.h"
#include "cabs/size.h"

#undef shadow_attr
#undef border_attr
#undef border

class Widget
{
	protected:
		WINDOW*     m_win_border;
		WINDOW*     m_win_shadow;
		std::string m_label;
		Position    m_position;
		Size	    m_size;
		bool        m_border = true;
		bool        m_shadow = false;
		bool        m_active = true;
		int         m_label_attr  = 0;
		int         m_border_attr = 0;
		int         m_shadow_attr = 0;

	public:
		Widget(void);
		virtual ~Widget(void);

	public:
		void attatch_to_window(WINDOW* win);
		virtual void draw(void) const;

	// GETTERS
	public:
		const Position&    position   (void) const;
		const Size&        size       (void) const;
		const std::string& label      (void) const;
		      bool         border        (void) const;
		      bool         shadow     (void) const;
		      bool         active     (void) const;
		      int          label_attr (void) const;
		      int          border_attr   (void) const;
		      int          shadow_attr(void) const;

	// SETTERS
	public:
		void position   (const Position&    new_position  );
		void size       (const Size&        new_size      );
		void label      (const std::string& new_label     );
		void border        (      bool         new_border       );
		void shadow     (      bool         new_shadow    );
		void active     (      bool         new_active    );
		void label_attr (      int         new_label_attr );
		void border_attr   (      int         new_border_attr   );
		void shadow_attr(      int         new_shadow_attr);
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
inline bool Widget::border(void) const
{
	return m_border;
}
inline bool Widget::shadow(void) const
{
	return m_shadow;
}
inline bool Widget::active(void) const
{
	return m_active;
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
inline void Widget::border(bool new_border)
{
	m_border = new_border;
}
inline void Widget::shadow(bool new_shadow)
{
	m_shadow = new_shadow;
}
inline void Widget::active(bool new_active)
{
	m_active = new_active;
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
