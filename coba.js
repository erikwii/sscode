var human = function(name, age, sex) {
    this.name   = name;
    this.age    = age;
    this.sex    = sex;

    return "My name is " + this.name + ", " + this.age + " years old. And I am a " + this.sex + ".";
}

console.log(human("Erik", 23, "boy"));