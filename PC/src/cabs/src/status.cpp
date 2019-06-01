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
	m_win = derwin(win, 1, COLS, LINES - 2, 0);
	if (m_win == NULL) exit(1);
}


void Status::redraw(void)
{
	werase(m_win);

	draw();
}

void Status::draw(void)
{
	wattron(m_win, m_mode_attr);
	wprintw(m_win, " %s ", Cabs::mode_to_c_str(m_mode));
	wattroff(m_win, m_mode_attr);

	wrefresh(m_win);
}
