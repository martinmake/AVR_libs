#include "screen.h"

Screen::Screen(int x, int y, int w, int h)
	: m_x(x), m_y(y), m_w(w), m_h(h)
{
	m_win = newwin(m_h, m_w, m_y, m_x);
}

Screen::Screen(void)
{
	initscr();
	cbreak();
	keypad(stdscr, true);
	noecho();
	curs_set(0);

	m_win = stdscr;
}

Screen::~Screen(void)
{
	if (m_win == stdscr)
		endwin();
	else
		delwin(m_win);
}

void Screen::redraw(void)
{
	werase(m_win);

	std::list<Widget>::iterator widget_it       = m_widgets.begin();
	std::list<bool>::iterator   widget_state_it = m_widget_states.begin();
	for (; widget_it != m_widgets.end(); widget_it++, widget_state_it++) {
		if (*widget_state_it)
			widget_it->draw();
	}

	wrefresh(m_win);
}

Screen& Screen::operator<<(Widget& widget)
{
	widget.attatch_to_window(m_win);
	m_widgets.push_back(widget);
	m_widget_states.push_back(true);

	return *this;
}
