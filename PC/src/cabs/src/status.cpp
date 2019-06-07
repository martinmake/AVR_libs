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
	m_parent_win = win;
	m_win = derwin(m_parent_win, 1, getmaxx(m_parent_win) - 1, getmaxy(m_parent_win) - 2, 0);
}

void Status::resize(void)
{
	delwin(m_win);
	m_win = derwin(m_parent_win, 1, getmaxx(m_parent_win) - 1, getmaxy(m_parent_win) - 2, 0);
	draw();
}


void Status::redraw(void) const
{
	erase();

	draw();
}

void Status::draw(void) const
{
	wattron (m_win, m_mode_attr);
	wprintw(m_win, " %s ", Cabs::mode_to_c_str(m_mode));
	wattroff(m_win, m_mode_attr);

	wrefresh(m_win);
}
