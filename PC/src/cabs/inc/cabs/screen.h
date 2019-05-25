#ifndef _CABS_SCREEN_H_
#define _CABS_SCREEN_H_

#include <ncurses.h>
#include <memory>
#include <vector>

#include "cabs/widget.h"

class Screen
{
	private:
		std::vector<std::shared_ptr<Widget>> m_widgets;

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
		template <typename W>
		Screen& operator<<(W& widget);

	// HANDLERS
	public:
		virtual void handle_key(int key);

	// OPERATORS
	public:
		Widget& operator[](int index);
};

template <typename W>
Screen& Screen::operator<<(W& widget)
{
	widget.attatch_to_window(m_win);
	m_widgets.push_back(std::make_shared<W>(widget));

	return *this;
}

#include "application.h"

#endif
