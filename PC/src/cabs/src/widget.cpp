#include "widget.h"

Widget::Widget(void)
{
}

Widget::~Widget(void)
{
	delwin(m_win_box);
	delwin(m_win_shadow);
}

void Widget::attatch_to_window(WINDOW* win)
{
	int x, y;

	x = Position::translate_x(win, m_position.x(), m_size.w());
	y = Position::translate_y(win, m_position.y(), m_size.h());

	m_win_box        = derwin(win, m_size.h(), m_size.w(), y,     x    );
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

	if (m_box)
	{
		wattr_on(m_win_box, m_box_attr, NULL);
		wborder(m_win_box, 0, 0, 0, 0, 0, 0, 0, 0);
		wattr_off(m_win_box, m_box_attr, NULL);
		if (!m_label.empty())
		{
			wmove(m_win_box, 0, 2);
			waddch(m_win_box, '|' | m_box_attr);
			wprintw(m_win_box, " %s ", m_label.c_str());
			waddch(m_win_box, '|' | m_box_attr);
		}
	}

	if (!m_label.empty())
	{
		wmove(m_win_box, 0, 3);
		wattr_on(m_win_box, m_label_attr, NULL);
		wprintw(m_win_box, " %s ", m_label.c_str());
		wattr_off(m_win_box, m_label_attr, NULL);
	}

	wrefresh(m_win_box);
	wrefresh(m_win_shadow);
}
