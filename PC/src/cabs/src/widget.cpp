#include <ncurses.h>
#include <dialog.h>

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
	if (!m_is_visible)
		return;

	if (m_is_shadowed)
	{
		wattr_on(m_win_shadow, m_shadow_attr, NULL);
		mvwaddch(m_win_shadow, m_size.h() - 1, 0, ACS_LLCORNER);
		whline(m_win_shadow, 0, m_size.w());
		mvwaddch(m_win_shadow, m_size.h() - 1, m_size.w() - 1, ACS_LRCORNER);
		mvwaddch(m_win_shadow, 0, m_size.w() - 1, ACS_URCORNER);
		mvwvline(m_win_shadow, 1, m_size.w() - 1, 0, m_size.h() - 2);
		wattr_off(m_win_shadow, m_shadow_attr, NULL);
	}

	if (m_is_bordered)
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

	if (m_is_selected)
	{
		wmove(m_win_border, 0, 0);
		waddch(m_win_border, '#' | A_STANDOUT | A_BLINK);
	}

	wrefresh(m_win_border);
	wrefresh(m_win_shadow);
}

void Widget::handle_key(int key)
{
	switch (key)
	{
		case ESC:
			m_is_selected = true;
			break;
	}
}
