#ifndef _CABS_SIZE_H_
#define _CABS_SIZE_H_

#include <ncurses.h>
#include <math.h>

class Size
{
	private:
		const WINDOW* m_parent_win   = nullptr;
		float   m_percentual_w = 0.0;
		float   m_percentual_h = 0.0;
		int     m_widget_gap   = 0;

	public:
		Size(float initial_w, float initial_h);
		Size(void);
		~Size(void);

	// GETTERS
	public:
		int   w(void) const;
		int   h(void) const;

	// SETTERS
	public:
		void w(float new_percentual_w);
		void h(float new_percentual_h);
		void parent_win(const WINDOW* new_parent_win);
		void widget_gap(int new_widget_gap);
};

// GETTERS
inline int Size::w(void) const
{
	return round(getmaxx(m_parent_win) * m_percentual_w) - 3 - (m_widget_gap ? 1 : 0) - m_widget_gap;
}
inline int Size::h(void) const
{
	return round(getmaxy(m_parent_win) * m_percentual_h) - 3 - m_widget_gap;
}

// SETTERS
inline void Size::w(float new_percentual_w)
{
	m_percentual_w = new_percentual_w;
}
inline void Size::h(float new_percentual_h)
{
	m_percentual_h = new_percentual_h;
}
inline void Size::parent_win(const WINDOW* new_parent_win)
{
	m_parent_win = new_parent_win;
}
inline void Size::widget_gap(int new_widget_gap)
{
	m_widget_gap = new_widget_gap;
}

#endif
