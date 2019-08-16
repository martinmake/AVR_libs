#ifndef _SML_H_
#define _SML_H_

#define DECLARATION_GETTER(getter_name, return_type) return_type getter_name(void                      ) const;
#define DECLARATION_SETTER(setter_name, arg_type   ) void        setter_name(arg_type new_##setter_name);

#define DECLARATION_CTOR_COPY(class_name) class_name(const class_name&  other);
#define DECLARATION_CTOR_MOVE(class_name) class_name(      class_name&& other);

#define DECLARATION_COPY(class_name) void copy(const class_name&  other);
#define DECLARATION_MOVE(class_name) void move(      class_name&& other);

#define DECLARATION_CTOR(class_name) class_name(void);

#define DECLARATION_DTOR        (class_name)         ~class_name(void);
#define DECLARATION_DTOR_VIRTUAL(class_name) virtual ~class_name(void);

#define DECLARATION_MANDATORY(class_name)          \
	public:                                    \
		DECLARATION_CTOR(class_name)       \
                                                   \
		DECLARATION_COPY_CTOR(class_name)  \
		DECLARATION_MOVE_CTOR(class_name)  \
                                                   \
		DECLARATION_DTOR(class_name)       \
                                                   \
	protected:                                 \
		DECLARATION_COPY(class_name)       \
		DECLARATION_MOVE(class_name)       \
	public:                                    \
		DECLARATION_COPY(class_name)       \
		DECLARATION_MOVE(class_name)       \
                                                   \
		DECLARATION_ASSIGNMENT(class_name) \
		DECLARATION_ASSIGNMENT(class_name)
#define DECLARATION_MANDATORY_DERIVE_ONLY(class_name) \
	protected:                                    \
		DECLARATION_CTOR(class_name)          \
        	                                      \
		DECLARATION_CTOR_COPY(class_name)     \
		DECLARATION_CTOR_MOVE(class_name)     \
		                                      \
		DECLARATION_DTOR_VIRTUAL(class_name)  \
		                                      \
		DECLARATION_COPY(class_name)          \
		DECLARATION_MOVE(class_name)

#define DEFINITION_DEFAULT_GETTER(class_name, getter_name, return_type) inline return_type class_name::getter_name(void                      ) const { return m_##getter_name;              }
#define DEFINITION_DEFAULT_SETTER(class_name, setter_name, arg_type   ) inline void        class_name::setter_name(arg_type new_##setter_name)       { m_##setter_name = new_##setter_name; }

#define DEFINITION_CTOR_COPY(class_name) inline class_name::class_name(const class_name&  other) : class_name() { copy(          other ); }
#define DEFINITION_CTOR_MOVE(class_name) inline class_name::class_name(      class_name&& other)                { move(std::move(other)); }

#define DEFINITION_ASSIGNMENT_OPERATOR_COPY(class_name) inline class_name& class_name::operator=(const class_name&  other) { copy(          other ); return *this; }
#define DEFINITION_ASSIGNMENT_OPERATOR_MOVE(class_name) inline class_name& class_name::operator=(      class_name&& other) { move(std::move(other)); return *this; }

#define DEFINITION_MANDATORY(class_name)          \
	DEFINITION_CTOR_COPY(Base)                \
	DEFINITION_CTOR_MOVE(Base)                \
	                                          \
	DEFINITION_ASSIGNMENT_OPERATOR_COPY(Base) \
	DEFINITION_ASSIGNMENT_OPERATOR_MOVE(Base)

#define DEFINITION_MANDATORY_DERIVE_ONLY(class_name) \
	DEFINITION_CTOR_COPY(Base)                   \
	DEFINITION_CTOR_MOVE(Base)


#endif
