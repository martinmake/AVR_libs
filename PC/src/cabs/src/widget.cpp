#include "widget.h"

Widget::Widget(void)
{
}

Widget::~Widget(void)
{
	delwin(m_win);
	delwin(m_win_shadow);
}

void Widget::attatch_to_window(WINDOW* win)
{
	int x, y;

	x = Position::translate_x(win, m_position.x(), m_size.w());
	y = Position::translate_y(win, m_position.y(), m_size.h());

	m_win        = derwin(win, m_size.h(), m_size.w(), y,     x    );
	m_win_shadow = derwin(win, m_size.h(), m_size.w(), y + 1, x + 1);
}

void Widget::draw(void) const
{
	if (!m_active)
		return;

	if (m_shadow)
		wborder(m_win_shadow, ' ', 0, ' ', 0, ' ', 0, 0, 0);

	if (m_box)
		wborder(m_win, 0, 0, 0, 0, 0, 0, 0, 0);

	if (!m_label.empty())
		mvwprintw(m_win, 0, 2, "| %s |", m_label.c_str());

	wrefresh(m_win);
	wrefresh(m_win_shadow);
}
