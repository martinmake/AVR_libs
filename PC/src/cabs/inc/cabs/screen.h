#ifndef _CABS_SCREEN_H_
#define _CABS_SCREEN_H_

#include <ncurses.h>
#include <memory>
#include <vector>

#include "cabs/widget.h"
#include "cabs/cabs.h"

class Screen
{
	private:
		std::vector<std::shared_ptr<Widget>> m_widgets;
		            std::shared_ptr<Widget>  m_selected_widget;

	protected:
		WINDOW*     m_win;
		Position    m_position;
		Size	    m_size;
		std::string m_name;

	public:
		Screen(const Position& position, const Size& size);
		Screen(void);
		virtual ~Screen(void);

	public:
		void draw(void) const;
		void redraw(void) const;
		void move(Cabs::Direction direction);

	// HANDLERS
	public:
		virtual void handle_key(int key);

	// OPERATORS
	public:
		Widget& operator[](int index);
		template <typename W>
		Screen& operator<<(W& widget);
};

template <typename W>
Screen& Screen::operator<<(W& widget)
{
	widget.attatch_to_window(m_win);
	m_widgets.push_back(std::make_shared<W>(widget));

	if (m_selected_widget == nullptr)
	{
		m_selected_widget = m_widgets.front();
		m_selected_widget->is_selected(true);
	}

	return *this;
}

#include "application.h"

#endif
