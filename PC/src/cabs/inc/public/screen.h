#ifndef _CABS_SCREEN_H_
#define _CABS_SCREEN_H_

#include <ncurses.h>
#include <memory>
#include <vector>

#include "widget.h"

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
		Screen& operator<<(Widget& widget);

	// HANDLERS
	public:
		virtual void handle_key(int key);

	// OPERATORS
	public:
		Widget& operator[](int index);
};

#include "application.h"

#endif
