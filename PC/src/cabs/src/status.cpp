#include "cabs/status.h"
Status::Status(void)
{
}

Status::~Status(void)
{
	delwin(m_win);
}

void Status::attatch_to_window(WINDOW* win)
{
	m_win = derwin(win, 1, getmaxx(win) - 1, getmaxy(win) - 1, 0);
}

void Status::resize(void)
{
	wresize(m_win, 1, getmaxx(m_win) - 1);
	mvwin  (m_win, getmaxy(m_win) - 1, 0);
	draw();
}


void Status::redraw(void) const
{
	werase(m_win);

	draw();
}

void Status::draw(void) const
{
	wattron (m_win, m_mode_attr);
	wprintw(m_win, " %s ", Cabs::mode_to_c_str(m_mode));
	wattroff(m_win, m_mode_attr);

	wrefresh(m_win);
}
