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
		WINDOW*     m_win        = nullptr;
		WINDOW*     m_border_win = nullptr;
		WINDOW*     m_shadow_win = nullptr;
		WINDOW*     m_parent_win = nullptr;
		std::string m_label;
		Position    m_position;
		Size        m_size;
		bool        m_is_bordered = false;
		bool        m_is_shadowed = false;
		bool        m_is_visible  = false;
		bool        m_is_selected = false;

	public:
		Widget(void);
		virtual ~Widget(void);

	public:
		void attatch_to_window(WINDOW* parent_win);
		virtual void clear(void)  const;
		virtual void resize(void);
		virtual void draw(void)   const;
		virtual void redraw(void) const;

	public:
		virtual void redraw_inside(void) const;
		virtual void draw_inside(void)   const;
		virtual void clear_inside(void)  const;
		virtual void resize_inside(void);

	// HANDLERS
	public:
		void handle_key(int key);

	// GETTERS
	public:
		const Position&    position   (void) const;
		const Size&        size       (void) const;
		const std::string& label      (void) const;
		      bool         is_bordered(void) const;
		      bool         is_shadowed(void) const;
		      bool         is_visible (void) const;
		      bool         is_selected(void) const;

	// SETTERS
	public:
		void position   (const Position&    new_position   );
		void size       (const Size&        new_size       );
		void label      (const std::string& new_label      );
		void is_bordered(      bool         new_is_bordered);
		void is_shadowed(      bool         new_is_shadowed);
		void is_visible (      bool         new_is_visible );
		void is_selected(      bool         new_is_selected);
		void widget_gap (      int          new_widget_gap );
};

inline void Widget::redraw(void) const
{
	clear();
	draw();
}

inline void Widget::resize_inside(void)
{
	draw();
}

inline void Widget::redraw_inside(void) const
{
	clear_inside();
	draw_inside();
}

inline void Widget::clear(void) const
{
	if (m_border_win) werase(m_border_win);
	if (m_shadow_win) werase(m_shadow_win);
	clear_inside();
}

// GETTERS
inline const Position& Widget::position(void) const { return m_position;    }
inline const Size& Widget::size        (void) const { return m_size;        }
inline const std::string& Widget::label(void) const { return m_label;       }
inline       bool Widget::is_bordered  (void) const { return m_is_bordered; }
inline       bool Widget::is_shadowed  (void) const { return m_is_shadowed; }
inline       bool Widget::is_visible   (void) const { return m_is_visible;  }
inline       bool Widget::is_selected  (void) const { return m_is_selected; }

// SETTERS
inline void Widget::position   (const Position& new_position) { m_position    = new_position;      }
inline void Widget::size       (const Size& new_size        ) { m_size        = new_size;          }
inline void Widget::label      (const std::string& new_label) { m_label       = new_label;         }
inline void Widget::is_bordered(      bool new_is_bordered  ) { m_is_bordered = new_is_bordered;   }
inline void Widget::is_shadowed(      bool new_is_shadowed  ) { m_is_shadowed = new_is_shadowed;   }
inline void Widget::is_visible (      bool new_is_visible   ) { m_is_visible  = new_is_visible;    }
inline void Widget::is_selected(      bool new_is_selected  ) { m_is_selected = new_is_selected;   }
inline void Widget::widget_gap (      int new_widget_gap    ) { m_size.widget_gap(new_widget_gap); }

#endif
