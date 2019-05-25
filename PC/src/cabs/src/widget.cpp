#include <ncurses.h>

#include "cabs/widget.h"

Widget::Widget(void)
{
}

Widget::~Widget(void)
{
	delwin(m_win_border);
	delwin(m_win_shadow);
}

void Widget::attatch_to_window(WINDOW* win)
{
	int x, y;

	x = Position::translate_x(win, m_position.x(), m_size.w());
	y = Position::translate_y(win, m_position.y(), m_size.h());

	m_win_border = derwin(win, m_size.h(), m_size.w(), y,     x    );
	m_win_shadow = derwin(win, m_size.h(), m_size.w(), y + 1, x + 1);
}

void Widget::draw(void) const
{
	if (!m_active)
		return;

	if (m_shadow)
	{
		wattr_on(m_win_shadow, m_shadow_attr, NULL);
		wborder(m_win_shadow, ' ', 0, ' ', 0, ' ', 0, 0, 0);
		wattr_off(m_win_shadow, m_shadow_attr, NULL);
	}

	if (m_border)
	{
		wattr_on(m_win_border, m_border_attr, NULL);
		wborder(m_win_border, 0, 0, 0, 0, 0, 0, 0, 0);
		wattr_off(m_win_border, m_border_attr, NULL);
		if (!m_label.empty())
		{
			wmove(m_win_border, 0, 2);
			waddch(m_win_border, '|' | m_border_attr);
			wmove(m_win_border, 0, 5 + m_label.size());
			waddch(m_win_border, '|' | m_border_attr);
		}
	}

	if (!m_label.empty())
	{
		wmove(m_win_border, 0, 3);
		wattr_on(m_win_border, m_label_attr, NULL);
		wprintw(m_win_border, " %s ", m_label.c_str());
		wattr_off(m_win_border, m_label_attr, NULL);
	}

	wrefresh(m_win_border);
	wrefresh(m_win_shadow);
}
