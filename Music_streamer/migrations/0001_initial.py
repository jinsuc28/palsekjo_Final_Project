# Generated by Django 3.2.9 on 2021-11-26 01:53

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='UserSong',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('emotion', models.CharField(max_length=100)),
                ('song', models.CharField(max_length=250)),
            ],
        ),
    ]