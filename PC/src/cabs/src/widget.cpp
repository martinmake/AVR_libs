#include <ncurses.h>
#include <dialog.h>

#include "cabs/widget.h"
#include "cabs/application.h"

Widget::Widget(void)
{
}

Widget::~Widget(void)
{
	delwin(m_border_win);
	delwin(m_shadow_win);
}

void Widget::attatch_to_window(WINDOW* win)
{
	int x, y;

	x = m_position.x(win, m_size.w());
	y = m_position.y(win, m_size.h());

	m_border_win = derwin(win, m_size.h(), m_size.w(), y,     x    );
	m_shadow_win = derwin(win, m_size.h(), m_size.w(), y + 1, x + 1);
}

void Widget::clear_inside(void) const
{
	if (!m_is_visible)
		return;

	wbkgd(m_border_win, application.widget_background_attr());

	wmove(m_border_win, 0, 0);
	for (uint8_t i = m_size.h() - 1; i; i--)
		waddch(m_border_win, '\n');
}

void Widget::draw_inside(void) const
{
}

void Widget::draw(void) const
{

	clear_inside();

	draw_inside();

	if (m_is_shadowed)
	{
		wattr_on(m_shadow_win, application.shadow_attr(), NULL);
		mvwaddch(m_shadow_win, m_size.h() - 1, 0, ACS_LLCORNER);
		whline(m_shadow_win, 0, m_size.w());
		mvwaddch(m_shadow_win, m_size.h() - 1, m_size.w() - 1, ACS_LRCORNER);
		mvwaddch(m_shadow_win, 0, m_size.w() - 1, ACS_URCORNER);
		mvwvline(m_shadow_win, 1, m_size.w() - 1, 0, m_size.h() - 2);
		wattr_off(m_shadow_win, application.shadow_attr(), NULL);
	}

	if (m_is_bordered)
	{
		wattr_on(m_border_win, application.border_attr(), NULL);
		wborder(m_border_win, 0, 0, 0, 0, 0, 0, 0, 0);
		wattr_off(m_border_win, application.border_attr(), NULL);

		if (!m_label.empty())
		{
			wmove(m_border_win, 0, 2);
			waddch(m_border_win, '|' | application.border_attr());
			wmove(m_border_win, 0, 5 + m_label.size());
			waddch(m_border_win, '|' | application.border_attr());
		}
	}

	if (!m_label.empty())
	{
		wmove(m_border_win, 0, 3);
		wattr_on(m_border_win, application.label_attr(), NULL);
		wprintw(m_border_win, " %s ", m_label.c_str());
		wattr_off(m_border_win, application.label_attr(), NULL);
	}

	if (m_is_selected)
	{
		mvwaddch(m_border_win, 0,              0,              application.selected_attr());
		mvwaddch(m_border_win, 0,              m_size.w() - 1, application.selected_attr());
		mvwaddch(m_border_win, m_size.h() - 1, 0,              application.selected_attr());
		mvwaddch(m_border_win, m_size.h() - 1, m_size.w() - 1, application.selected_attr());
	}

	wrefresh(m_border_win);
	wrefresh(m_shadow_win);
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
