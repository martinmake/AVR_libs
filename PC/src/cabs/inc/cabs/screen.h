#ifndef _CABS_SCREEN_H_
#define _CABS_SCREEN_H_

#include <ncurses.h>
#include <memory>
#include <vector>

#include "cabs/widget.h"
#include "cabs/cabs.h"
#include "cabs/status.h"

class Screen
{
	private:
		std::vector<std::shared_ptr<Widget>> m_widgets;
		            std::shared_ptr<Widget>  m_selected_widget;

	private:
		Status m_status;

	public:
		int m_widget_gap;

	protected:
		WINDOW*     m_win;
		// Position    m_position;
		// Size	    m_size;
		// std::string m_name;

	public:
		Screen(void);
		virtual ~Screen(void);

	public:
		void draw(void) const;
		void redraw(void) const;
		void resize(void);
		void move(Cabs::Direction direction);

	// SETTERS
		void widget_gap(int new_widget_gap);

	// HANDLERS
	public:
		virtual void handle_key(int key);

	// OPERATORS
	public:
		Widget& operator[](int index);
		template <typename W>
		Screen& operator<<(W& widget);
};

inline void Screen::redraw(void) const
{
	werase(m_win);

	draw();
}

// SETTERS
inline void Screen::widget_gap(int new_widget_gap)
{
	m_widget_gap = new_widget_gap;
}

// OPERATORS
template <typename W>
Screen& Screen::operator<<(W& widget)
{
	widget.attatch_to_window(m_win);
	widget.widget_gap(m_widget_gap);
	m_widgets.push_back(std::make_shared<W>(widget));

	if (m_selected_widget == nullptr)
	{
		m_selected_widget = m_widgets.front();
		m_selected_widget->is_selected(true);
	}

	return *this;
}

#endif
