#include <ncurses.h>
#include <dialog.h>
#include <assert.h>

#include "cabs/widget.h"
#include "cabs/application.h"

Widget::Widget(void)
{
}

Widget::~Widget(void)
{
	if (m_win       ) delwin(m_win);
	if (m_border_win) delwin(m_border_win);
	if (m_shadow_win) delwin(m_shadow_win);

	m_win        = nullptr;
	m_border_win = nullptr;
	m_shadow_win = nullptr;
}

void Widget::attatch_to_window(WINDOW* parent_win)
{
	int x, y;
	int w, h;

	m_parent_win = parent_win;

	m_position.parent_win(m_parent_win);
	m_size.parent_win(m_parent_win);

	x = m_position.x();
	y = m_position.y();
	w = m_size.w();
	h = m_size.h();

	m_win        = derwin(m_parent_win, h,     w,     y,     x    );
	m_border_win = derwin(m_parent_win, h + 2, w + 2, y - 1, x - 1);
	m_shadow_win = derwin(m_parent_win, h + 2, w + 2, y,     x    );
}

void Widget::resize(void)
{
	int x, y;
	int w, h;

	x = m_position.x();
	y = m_position.y();
	w = m_size.w();
	h = m_size.h();

	delwin(m_win       );
	delwin(m_border_win);
	delwin(m_shadow_win);

	m_win        = derwin(m_parent_win, h,     w,     y,     x    );
	m_border_win = derwin(m_parent_win, h + 2, w + 2, y - 1, x - 1);
	m_shadow_win = derwin(m_parent_win, h + 2, w + 2, y,     x    );

	resize_inside();
}

void Widget::clear_inside(void) const
{
	if (!m_is_visible)
		return;

	wbkgd(m_win, application.widget_background_attr());

	wmove(m_win, 0, 0);
	for (uint8_t i = m_size.h() - 1; i; i--)
		waddch(m_win, '\n');
}

void Widget::draw_inside(void) const
{
}

void Widget::draw(void) const
{
	int w, h;

	w = m_size.w();
	h = m_size.h();

	redraw_inside();

	if (m_is_shadowed)
	{
		wattron(m_shadow_win, application.shadow_attr());
		mvwaddch(m_shadow_win, 0,     w + 1, ACS_URCORNER);
		mvwvline(m_shadow_win, 1,     w + 1, ACS_VLINE, h + 1);
		mvwaddch(m_shadow_win, h + 1, w + 1, ACS_LRCORNER);
		mvwhline(m_shadow_win, h + 1, 1,     ACS_HLINE, w);
		mvwaddch(m_shadow_win, h + 1, 0,     ACS_LLCORNER);
		wattroff(m_shadow_win, application.shadow_attr());
	}

	if (m_is_bordered)
	{
		wattron(m_border_win, application.border_attr());
		wborder(m_border_win, 0, 0, 0, 0, 0, 0, 0, 0);
		wattroff(m_border_win, application.border_attr());

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
		wattron(m_border_win, application.label_attr());
		wprintw(m_border_win, " %s ", m_label.c_str());
		wattroff(m_border_win, application.label_attr());
	}

	if (m_is_selected)
	{
		mvwaddch(m_border_win, 0,     0,     application.selected_attr());
		mvwaddch(m_border_win, 0,     w + 1, application.selected_attr());
		mvwaddch(m_border_win, h + 1, 0,     application.selected_attr());
		mvwaddch(m_border_win, h + 1, w + 1, application.selected_attr());
	}

	wrefresh(m_border_win);
	wrefresh(m_shadow_win);
}

void Widget::handle_key(int key)
{
	(void) key;
}
